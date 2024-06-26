package com.ssafy.backend.account.service;

import com.ssafy.backend.account.model.domain.vo.TokenVo;
import com.ssafy.backend.member.model.vo.GetMemberInformationVo;
import com.ssafy.backend.member.service.MemberFacade;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.HashMap;
import java.util.Map;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class AccountFacade {

    @Autowired
    AccountService accountService;

    @Autowired
    MemberFacade memberFacade;

    @Transactional
    public Map<String, Object> OAuthLogin(String memberCode, char loginType){
        TokenVo tokenVo = accountService.OAuthLogin(memberCode, loginType);

        GetMemberInformationVo getMemberInformation = memberFacade.getMemberInformation(tokenVo.getMemberSeq());

        Map<String, Object> resultMap = new HashMap<>();
        resultMap.put("tokenVo", tokenVo);
        resultMap.put("memberInformation", getMemberInformation);

        return resultMap;
    }
}
